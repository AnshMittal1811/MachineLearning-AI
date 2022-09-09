using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ControlCamera : MonoBehaviour
{
    public byte[] img = new byte[8];

    float dt = 0.1f;
    float dr = 1.0f;

    // Use this for initialization
    void Start()
    {
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        try
        {
            if (Input.GetKey("w"))
            {
                transform.Translate(0f, 0f, dt); // front
            }
            else if (Input.GetKey("s"))
            {
                transform.Translate(0f, 0f, -dt); // back
            }
            else if (Input.GetKey("d"))
            {
                transform.Translate(dt, 0f, 0f); // right
            }
            else if (Input.GetKey("a"))
            {
                transform.Translate(-dt, 0f, 0f); // left
            }
            else if (Input.GetKey("space"))
            {
                transform.Translate(0f, dt, 0f); // top
            }
            else if (Input.GetKey("return"))
            {
                transform.Translate(0f, -dt, 0f); // botom
            }
            else if (Input.GetKey("q"))
            {
                transform.Rotate(0f, dr, 0f); // yaw++
            }
            else if (Input.GetKey("e"))
            {
                transform.Rotate(0f, -dr, 0f); // yaw--
            }
            else if (Input.GetKey("z"))
            {
                transform.Rotate(dr, 0f, 0f); // pitch++
            }
            else if (Input.GetKey("x"))
            {
                transform.Rotate(-dr, 0f, 0f); // pitch--
            }
            else if (Input.GetKey("c"))
            {
                transform.Rotate(0f, 0f, dr); // roll++
            }
            else if (Input.GetKey("v"))
            {
                transform.Rotate(0f, 0f, -dr); // roll--
            }
            // for usability
            transform.rotation = Quaternion.Euler(transform.eulerAngles.x, transform.eulerAngles.y, 0.0f);

        }
        finally
        {
        }
    }
}